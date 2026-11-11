/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <args.h>
#include <autotune.h>
#include <densematrix.h>
#include <fasttext.h>
#include <MurmurHash3.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <real.h>
#include <vector.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;

using RealArray1D = nb::ndarray<nb::numpy, fasttext::real, nb::ndim<1>>;
using RealArray2D = nb::ndarray<nb::numpy, fasttext::real, nb::ndim<2>>;

nb::str castToPythonString(const std::string& s, const char* onUnicodeError) {
  PyObject* handle = PyUnicode_DecodeUTF8(s.data(), s.length(), onUnicodeError);
  if (!handle) {
    throw nb::python_error();
  }
  return nb::steal<nb::str>(handle);
}

std::vector<std::pair<fasttext::real, nb::str>> castToPythonString(
    const std::vector<std::pair<fasttext::real, std::string>>& predictions,
    const char* onUnicodeError) {
  std::vector<std::pair<fasttext::real, nb::str>> transformedPredictions;

  for (const auto& prediction : predictions) {
    transformedPredictions.emplace_back(
        prediction.first,
        castToPythonString(prediction.second, onUnicodeError));
  }

  return transformedPredictions;
}

std::pair<std::vector<nb::str>, std::vector<nb::str>> getLineText(
    fasttext::FastText& m,
    const std::string text,
    const char* onUnicodeError) {
  std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
  std::stringstream ioss(text);
  std::string token;
  std::vector<nb::str> words;
  std::vector<nb::str> labels;
  while (d->readWord(ioss, token)) {
    uint32_t h = d->hash(token);
    int32_t wid = d->getId(token, h);
    fasttext::entry_type type = wid < 0 ? d->getType(token) : d->getType(wid);

    if (type == fasttext::entry_type::word) {
      words.push_back(castToPythonString(token, onUnicodeError));
      // Labels must not be OOV!
    } else if (type == fasttext::entry_type::label && wid >= 0) {
      labels.push_back(castToPythonString(token, onUnicodeError));
    }
    if (token == fasttext::Dictionary::EOS)
      break;
  }
  return std::pair<std::vector<nb::str>, std::vector<nb::str>>(words, labels);
}

NB_MODULE(fastfloret_pybind, m) {
  nb::class_<fasttext::Args>(m, "args")
      .def(nb::init<>())
      .def_rw("input", &fasttext::Args::input)
      .def_rw("output", &fasttext::Args::output)
      .def_rw("lr", &fasttext::Args::lr)
      .def_rw("lrUpdateRate", &fasttext::Args::lrUpdateRate)
      .def_rw("dim", &fasttext::Args::dim)
      .def_rw("ws", &fasttext::Args::ws)
      .def_rw("epoch", &fasttext::Args::epoch)
      .def_rw("minCount", &fasttext::Args::minCount)
      .def_rw("minCountLabel", &fasttext::Args::minCountLabel)
      .def_rw("neg", &fasttext::Args::neg)
      .def_rw("wordNgrams", &fasttext::Args::wordNgrams)
      .def_rw("loss", &fasttext::Args::loss)
      .def_rw("model", &fasttext::Args::model)
      .def_rw("bucket", &fasttext::Args::bucket)
      .def_rw("minn", &fasttext::Args::minn)
      .def_rw("maxn", &fasttext::Args::maxn)
      .def_rw("mode", &fasttext::Args::mode)
      .def_rw("hashCount", &fasttext::Args::hashCount)
      .def_rw("thread", &fasttext::Args::thread)
      .def_rw("t", &fasttext::Args::t)
      .def_rw("label", &fasttext::Args::label)
      .def_rw("verbose", &fasttext::Args::verbose)
      .def_rw("pretrainedVectors", &fasttext::Args::pretrainedVectors)
      .def_rw("saveOutput", &fasttext::Args::saveOutput)
      .def_rw("seed", &fasttext::Args::seed)

      .def_rw("qout", &fasttext::Args::qout)
      .def_rw("retrain", &fasttext::Args::retrain)
      .def_rw("qnorm", &fasttext::Args::qnorm)
      .def_rw("cutoff", &fasttext::Args::cutoff)
      .def_rw("dsub", &fasttext::Args::dsub)

      .def_rw(
          "autotuneValidationFile", &fasttext::Args::autotuneValidationFile)
      .def_rw("autotuneMetric", &fasttext::Args::autotuneMetric)
      .def_rw(
          "autotunePredictions", &fasttext::Args::autotunePredictions)
      .def_rw("autotuneDuration", &fasttext::Args::autotuneDuration)
      .def_rw("autotuneModelSize", &fasttext::Args::autotuneModelSize)
      .def("setManual", [](fasttext::Args& m, const std::string& argName) {
        m.setManual(argName);
      });

  nb::enum_<fasttext::model_name>(m, "model_name")
      .value("cbow", fasttext::model_name::cbow)
      .value("skipgram", fasttext::model_name::sg)
      .value("supervised", fasttext::model_name::sup);

  nb::enum_<fasttext::loss_name>(m, "loss_name")
      .value("hs", fasttext::loss_name::hs)
      .value("ns", fasttext::loss_name::ns)
      .value("softmax", fasttext::loss_name::softmax)
      .value("ova", fasttext::loss_name::ova);

  nb::enum_<fasttext::metric_name>(m, "metric_name")
      .value("f1score", fasttext::metric_name::f1score)
      .value("f1scoreLabel", fasttext::metric_name::f1scoreLabel)
      .value("precisionAtRecall", fasttext::metric_name::precisionAtRecall)
      .value(
          "precisionAtRecallLabel",
          fasttext::metric_name::precisionAtRecallLabel)
      .value("recallAtPrecision", fasttext::metric_name::recallAtPrecision)
      .value(
          "recallAtPrecisionLabel",
          fasttext::metric_name::recallAtPrecisionLabel);

  nb::enum_<fasttext::mode_name>(m, "mode_name")
      .value("fasttext", fasttext::mode_name::fasttext)
      .value("floret", fasttext::mode_name::floret);
      // not exported into the parent scope because the names clash

  m.def(
      "train",
      [](fasttext::FastText& ft, fasttext::Args& a) {
        if (a.hasAutotune()) {
          fasttext::Autotune autotune(std::shared_ptr<fasttext::FastText>(
              &ft, [](fasttext::FastText*) {}));
          autotune.train(a);
        } else {
          ft.train(a);
        }
      },
      nb::call_guard<nb::gil_scoped_release>());

  nb::class_<fasttext::Vector>(m, "Vector")
      .def(nb::init<int64_t>())
      .def(
          "numpy",
          [](fasttext::Vector& v) {
            return RealArray1D(v.data(), {static_cast<size_t>(v.size())});
          },
          nb::rv_policy::reference_internal,
          "Return a NumPy view of the vector's data.");

  nb::class_<fasttext::DenseMatrix>(m, "DenseMatrix")
      .def(nb::init<>())
      .def(nb::init<int64_t, int64_t>())
      .def(
          "numpy",
          [](fasttext::DenseMatrix& dm) {
            return RealArray2D(
                dm.data(),
                {static_cast<size_t>(dm.size(0)),
                 static_cast<size_t>(dm.size(1))});
          },
          nb::rv_policy::reference_internal,
          "Return a NumPy view of the matrix's data.");

  nb::class_<fasttext::Meter>(m, "Meter")
      .def(nb::init<bool>())
      .def("scoreVsTrue", &fasttext::Meter::scoreVsTrue)
      .def(
          "precisionRecallCurveLabel",
          (std::vector<std::pair<double, double>>(fasttext::Meter::*)(int32_t)
               const) &
              fasttext::Meter::precisionRecallCurve)
      .def(
          "precisionRecallCurve",
          (std::vector<std::pair<double, double>>(fasttext::Meter::*)() const) &
              fasttext::Meter::precisionRecallCurve)
      .def(
          "precisionAtRecallLabel",
          (double (fasttext::Meter::*)(int32_t, double) const) &
              fasttext::Meter::precisionAtRecall)
      .def(
          "precisionAtRecall",
          (double (fasttext::Meter::*)(double) const) &
              fasttext::Meter::precisionAtRecall)
      .def(
          "recallAtPrecisionLabel",
          (double (fasttext::Meter::*)(int32_t, double) const) &
              fasttext::Meter::recallAtPrecision)
      .def(
          "recallAtPrecision",
          (double (fasttext::Meter::*)(double) const) &
              fasttext::Meter::recallAtPrecision);

  nb::class_<fasttext::FastText>(m, "fasttext")
      .def(nb::init<>())
      .def("getArgs", &fasttext::FastText::getArgs)
      .def(
          "getInputMatrix",
          [](fasttext::FastText& m) {
            std::shared_ptr<const fasttext::DenseMatrix> mm =
                m.getInputMatrix();
            return const_cast<fasttext::DenseMatrix*>(mm.get());
          },
          nb::rv_policy::reference)
      .def(
          "getOutputMatrix",
          [](fasttext::FastText& m) {
            std::shared_ptr<const fasttext::DenseMatrix> mm =
                m.getOutputMatrix();
            return const_cast<fasttext::DenseMatrix*>(mm.get());
          },
          nb::rv_policy::reference)
      .def(
          "setMatrices",
          [](fasttext::FastText& m,
             nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                 inputMatrix,
             nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                 outputMatrix) {
            // DenseMatrix copies the data, so the arrays need not outlive it
            m.setMatrices(
                std::make_shared<fasttext::DenseMatrix>(
                    inputMatrix.shape(0),
                    inputMatrix.shape(1),
                    const_cast<float*>(inputMatrix.data())),
                std::make_shared<fasttext::DenseMatrix>(
                    outputMatrix.shape(0),
                    outputMatrix.shape(1),
                    const_cast<float*>(outputMatrix.data())));
          })
      .def(
          "loadModel",
          [](fasttext::FastText& m, std::string s) { m.loadModel(s); })
      .def(
          "saveModel",
          [](fasttext::FastText& m, std::string s) { m.saveModel(s); })
      .def(
          "saveVectors",
          [](fasttext::FastText& m, std::string s) { m.saveVectors(s); })
      .def(
          "saveFloretVectors",
          [](fasttext::FastText& m, std::string s) { m.saveFloretVectors(s); })
      .def(
          "test",
          [](fasttext::FastText& m,
             const std::string& filename,
             int32_t k,
             fasttext::real threshold) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(false);
            m.test(ifs, k, threshold, meter);
            ifs.close();
            return std::tuple<int64_t, double, double>(
                meter.nexamples(), meter.precision(), meter.recall());
          })
      .def(
          "getMeter",
          [](fasttext::FastText& m, const std::string& filename, int32_t k) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(true);
            m.test(ifs, k, 0.0, meter);
            ifs.close();

            return meter;
          })
      .def(
          "getSentenceVector",
          [](fasttext::FastText& m,
             fasttext::Vector& v,
             const std::string text) {
            std::stringstream ioss(text);
            m.getSentenceVector(ioss, v);
          })
      .def(
          "tokenize",
          [](fasttext::FastText& m, const std::string text) {
            std::vector<std::string> text_split;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::stringstream ioss(text);
            std::string token;
            while (!ioss.eof()) {
              while (d->readWord(ioss, token)) {
                text_split.push_back(token);
              }
            }
            return text_split;
          })
      .def("getLine", &getLineText)
      .def(
          "multilineGetLine",
          [](fasttext::FastText& m,
             const std::vector<std::string> lines,
             const char* onUnicodeError) {
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::vector<std::vector<nb::str>> all_words;
            std::vector<std::vector<nb::str>> all_labels;
            for (const auto& text : lines) {
              auto pair = getLineText(m, text, onUnicodeError);
              all_words.push_back(pair.first);
              all_labels.push_back(pair.second);
            }
            return std::pair<
                std::vector<std::vector<nb::str>>,
                std::vector<std::vector<nb::str>>>(all_words, all_labels);
          })
      .def(
          "getVocab",
          [](fasttext::FastText& m, const char* onUnicodeError) {
            std::vector<nb::str> vocab_list;
            std::vector<int64_t> vocab_freq;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            vocab_freq = d->getCounts(fasttext::entry_type::word);
            for (int32_t i = 0; i < vocab_freq.size(); i++) {
              vocab_list.push_back(
                  castToPythonString(d->getWord(i), onUnicodeError));
            }
            return std::pair<std::vector<nb::str>, std::vector<int64_t>>(
                vocab_list, vocab_freq);
          })
      .def(
          "getLabels",
          [](fasttext::FastText& m, const char* onUnicodeError) {
            std::vector<nb::str> labels_list;
            std::vector<int64_t> labels_freq;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            labels_freq = d->getCounts(fasttext::entry_type::label);
            for (int32_t i = 0; i < labels_freq.size(); i++) {
              labels_list.push_back(
                  castToPythonString(d->getLabel(i), onUnicodeError));
            }
            return std::pair<std::vector<nb::str>, std::vector<int64_t>>(
                labels_list, labels_freq);
          })
      .def(
          "quantize",
          [](fasttext::FastText& m,
             const std::string input,
             bool qout,
             int32_t cutoff,
             bool retrain,
             int epoch,
             double lr,
             int thread,
             int verbose,
             int32_t dsub,
             bool qnorm) {
            fasttext::Args qa = fasttext::Args();
            qa.input = input;
            qa.qout = qout;
            qa.cutoff = cutoff;
            qa.retrain = retrain;
            qa.epoch = epoch;
            qa.lr = lr;
            qa.thread = thread;
            qa.verbose = verbose;
            qa.dsub = dsub;
            qa.qnorm = qnorm;
            m.quantize(qa);
          })
      .def(
          "predict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](fasttext::FastText& m,
             const std::string text,
             int32_t k,
             fasttext::real threshold,
             const char* onUnicodeError) {
            std::stringstream ioss(text);
            std::vector<std::pair<fasttext::real, std::string>> predictions;
            m.predictLine(ioss, predictions, k, threshold);

            return castToPythonString(predictions, onUnicodeError);
          })
      .def(
          "multilinePredict",
          // NOTE: text needs to end in a newline
          // to exactly mimic the behavior of the cli
          [](fasttext::FastText& m,
             const std::vector<std::string>& lines,
             int32_t k,
             fasttext::real threshold,
             const char* onUnicodeError) {
            std::vector<RealArray1D> allProbabilities;
            std::vector<std::vector<nb::str>> allLabels;
            std::vector<std::pair<fasttext::real, std::string>> predictions;

            for (const std::string& text : lines) {
              std::stringstream ioss(text);
              m.predictLine(ioss, predictions, k, threshold);
              std::vector<nb::str> labels;

              const size_t n = predictions.size();
              fasttext::real* probabilities = new fasttext::real[n];
              nb::capsule owner(probabilities, [](void* p) noexcept {
                delete[] static_cast<fasttext::real*>(p);
              });
              for (size_t i = 0; i < n; i++) {
                probabilities[i] = predictions[i].first;
                labels.push_back(
                    castToPythonString(predictions[i].second, onUnicodeError));
              }

              allProbabilities.push_back(RealArray1D(probabilities, {n}, owner));
              allLabels.push_back(std::move(labels));
            }

            return std::make_pair(allLabels, allProbabilities);
          })
      .def(
          "testLabel",
          [](fasttext::FastText& m,
             const std::string filename,
             int32_t k,
             fasttext::real threshold) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
              throw std::invalid_argument("Test file cannot be opened!");
            }
            fasttext::Meter meter(false);
            m.test(ifs, k, threshold, meter);
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            std::unordered_map<std::string, nb::dict> returnedValue;
            for (int32_t i = 0; i < d->nlabels(); i++) {
              nb::dict scores;
              scores["precision"] = meter.precision(i);
              scores["recall"] = meter.recall(i);
              scores["f1score"] = meter.f1Score(i);
              returnedValue[d->getLabel(i)] = scores;
            }

            return returnedValue;
          })
      .def(
          "getWordId",
          [](fasttext::FastText& m, const std::string& word) {
            return m.getWordId(word);
          })
      .def(
          "getSubwordId",
          [](fasttext::FastText& m, const std::string word) {
            return m.getSubwordId(word);
          })
      .def(
          "getLabelId",
          [](fasttext::FastText& m, const std::string& label) {
            return m.getLabelId(label);
          })
      .def(
          "getInputVector",
          [](fasttext::FastText& m, fasttext::Vector& vec, int32_t ind) {
            m.getInputVector(vec, ind);
          })
      .def(
          "getWordVector",
          [](fasttext::FastText& m,
             fasttext::Vector& vec,
             const std::string word) { m.getWordVector(vec, word); })
      .def(
          "getNN",
          [](fasttext::FastText& m,
             const std::string& word,
             int32_t k,
             const char* onUnicodeError) {
            return castToPythonString(m.getNN(word, k), onUnicodeError);
          })
      .def(
          "getAnalogies",
          [](fasttext::FastText& m,
             const std::string& wordA,
             const std::string& wordB,
             const std::string& wordC,
             int32_t k,
             const char* onUnicodeError) {
            return castToPythonString(
                m.getAnalogies(k, wordA, wordB, wordC), onUnicodeError);
          })
      .def(
          "getSubwords",
          [](fasttext::FastText& m,
             const std::string word,
             const char* onUnicodeError) {
            std::vector<std::string> subwords;
            std::vector<int32_t> ngrams;
            std::shared_ptr<const fasttext::Dictionary> d = m.getDictionary();
            d->getSubwords(word, ngrams, subwords);
            std::vector<nb::str> transformedSubwords;

            for (const auto& subword : subwords) {
              transformedSubwords.push_back(
                  castToPythonString(subword, onUnicodeError));
            }

            return std::pair<std::vector<nb::str>, std::vector<int32_t>>(
                transformedSubwords, ngrams);
          })
      .def("isQuant", [](fasttext::FastText& m) { return m.isQuant(); });
}
