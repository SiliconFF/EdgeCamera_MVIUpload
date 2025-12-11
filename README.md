<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)][linkedin-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload">
    <img src="images/logo.png" alt="Logo" width="200" height="200">
  </a>

<h3 align="center">MVI Dragonfly</h3>

  <p align="center">
    Turn any computer into a lightweight edge camera that can upload directly to an MVI Edge device 
    eliminating time synchronization issues that can occur with large networks and RTSP streams.
    <br />
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload">View Demo</a>
    &middot;
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

For some backstory on this project. I have been implementing applications in assembly processes that use IBM MVI Edge to run quality inspection and object detection models. 
The single biggest hurdle that was encountered was the syncronizaiton of the time of trigger and the time of the RTSP frame that was captured. To more clearly layout the issue 
that was encounted please see the flow chart below:

[Chart to be added later]


This issue gave me the idea to implement an interface to upload photos taken on an edge device (like a raspberrypi) connected to the camera to "Image Folder" devices on the MVI Edge. With this implmentation the compute overhead in the MVI server is reduced and has been tested to vastly improve the time sync from sometimes being off by upwards of 4s to being within ~200ms. The main design goal is to give the integrators of MVI Edge more flexibility with what cameras they can use and to deal with networking
and financial restrictions on equipment usage.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/SiliconFF/EdgeCamera_MVIUpload.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin SiliconFF/EdgeCamera_MVIUpload
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This project is intended to turn any computer into an edge camera device with higher time syncronization over large networks than just a standard RTSP based stream.

### Example Scenario:

RTSP streaming in MVI Edge exhibits poor time synchronization between the photo capture request and the actual frame capture in real time. This issue is particularly problematic for time-sensitive applications, such as processes that wait for inspection completion or involve moving objects.
To address this, consider one of the following alternatives:

Connect the RTSP camera directly to a single-board computer (e.g., Raspberry Pi) or use a different camera interface, such as CSI or USB. (There are plans to add GigE support though this seems overkill for this application)

These options allow connection to your MVI Edge instance with continuous frame capture. The system will wait for a message on a designated MQTT trigger topic of your choice. Upon receiving the message, a frame is captured and uploaded to your specified image folder for inspection.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
- [ ] Add YAML schema 
- [ ] Verify functionality with RPI CSI modules
- [ ] Allow multiple devices to be set as a target
    - [ ] Devices can be linked to different trigger MQTT topics for more comprehensive coverage   
- [ ] Develop a supervisor webserver for management and config changes
- [ ] Support for reading from GigE cameras
- [ ] Prebuilt RPI install script to automatically configure a raspberry pi to be deployed


See the [open issues](https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SiliconFF/EdgeCamera_MVIUpload" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Collin Finetti - finetticj@gmail.com

Project Link: [https://github.com/SiliconFF/EdgeCamera_MVIUpload](https://github.com/SiliconFF/EdgeCamera_MVIUpload)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[contributors-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[forks-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/network/members
[stars-shield]: https://img.shields.io/github/stars/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[stars-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/stargazers
[issues-shield]: https://img.shields.io/github/issues/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[issues-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues
[license-shield]: https://img.shields.io/github/license/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[license-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/collin-finetti
[product-screenshot]: images/project_logo.png

<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
