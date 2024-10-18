Learning diffusion at lightspeed
================================

`Paper <https://arxiv.org/abs/2406.12616>`_ & `Code <https://github.com/antonioterpin/jkonet-star>`_

.. container:: full-width-container

   .. raw:: html

      <style>
        .full-width-container {
          display: flex;
          justify-content: space-between;
          flex-wrap: wrap;
        }

        .full-width-container img {
          width: 45%;
          margin-bottom: 10px;
        }

        @media (max-width: 768px) {
            .full-width-container {
               display: flex;
               justify-content: center;
               flex-wrap: wrap;
            }
            .full-width-container img {
               width: 90%;
            }
        }
      </style>

   .. image:: _static/cover.png
      :align: left
      :alt: Cover image

   .. image:: _static/preview.png
      :align: right
      :alt: Preview image

Diffusion regulates numerous natural processes and drives the dynamics of many successful generative models. 
Current models for learning diffusion terms from observational data often require complex bilevel optimization problems and primarily focus on modeling the drift component of the system.

We propose a new simple model, **JKOnet\***, which bypasses the complexity of existing architectures while presenting significantly enhanced representational capabilities: JKOnet\* recovers the potential, interaction, and internal energy components of the underlying diffusion process. JKOnet\* minimizes a simple quadratic loss and drastically outperforms other baselines in terms of sample efficiency, computational complexity, and accuracy. Additionally, JKOnet\* provides a closed-form optimal solution for linearly parametrized functionals, and, when applied to predict the evolution of cellular processes from real-world data, it achieves state-of-the-art accuracy at a fraction of the computational cost of all existing methods.

.. admonition:: Key advantages of JKOnet\*

   - **Outperforms** existing baselines in sample efficiency, computational complexity, and accuracy.
   - Learns the different components of the diffusion process, including potential, interaction, and internal energy.
   - Provides a **closed-form optimal solution** for linearly parametrized functionals.
   - Achieves **state-of-the-art accuracy** in predicting cellular process evolution at a fraction of the computational cost of existing methods.


.. table:: JKOnet\* vs JKOnet

   +-----------------+-------------------+---------------------+------------------+---------+
   |                 | Potential energy  | Interaction energy  | Internal energy  | Speed   |
   +-----------------+-------------------+---------------------+------------------+---------+
   | **JKOnet**      |        ✅         |     ❌              |   ❌             | slow    |
   +-----------------+-------------------+---------------------+------------------+---------+
   | **JKOnet\***    | ✅                |   ✅                |       ✅         | fast 🔥 |
   +-----------------+-------------------+---------------------+------------------+---------+


Our methodology is based on the interpretation of diffusion processes as energy-minimizing trajectories in the probability space via the so-called **JKO scheme**, which we study via its first-order optimality conditions.

Check out the `paper <https://arxiv.org/abs/2406.12616>`_ for an intuition as well as an in-depth explanation and thorough comparisons with existing methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pages/getting_started
   pages/benchmarks
   pages/developer_resources
   pages/applications

Citation 🙏
-----------

If you use this code in your research, please cite our paper (NeurIPS 2024, Oral Presentation):

.. code-block:: latex

   @article{terpin2024learning,
      title={Learning diffusion at lightspeed},
      author={Terpin, Antonio and Lanzetti, Nicolas and Gadea, Mart{\'\i}n and D\"{o}rfler, Florian},
      journal={Advances in Neural Information Processing Systems},
      volume={37},
      pages={6797--6832},
      year={2024}
   }


Contact and contributing
------------------------

If you have any questions, want to signal an error or contribute to the project, feel free to reach out to **Antonio Terpin** via email: `aterpin@ethz.ch <mailto:aterpin@ethz.ch>`_ or directly open an issue/PR on the `GitHub repository <http://github.com/antonioterpin/jkonet-star>`_.