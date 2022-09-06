# coding: utf-8

"""
    Speech Services API v3.0

    Speech Services API v3.0.  # noqa: E501

    OpenAPI spec version: v3.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from swagger_client.configuration import Configuration


class WebHook(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        "web_url": "str",
        "links": "WebHookLinks",
        "properties": "WebHookProperties",
        "_self": "str",
        "display_name": "str",
        "description": "str",
        "events": "WebHookEvents",
        "created_date_time": "datetime",
        "last_action_date_time": "datetime",
        "status": "Status",
        "custom_properties": "dict(str, str)",
    }

    attribute_map = {
        "web_url": "webUrl",
        "links": "links",
        "properties": "properties",
        "_self": "self",
        "display_name": "displayName",
        "description": "description",
        "events": "events",
        "created_date_time": "createdDateTime",
        "last_action_date_time": "lastActionDateTime",
        "status": "status",
        "custom_properties": "customProperties",
    }

    def __init__(
        self,
        web_url=None,
        links=None,
        properties=None,
        _self=None,
        display_name=None,
        description=None,
        events=None,
        created_date_time=None,
        last_action_date_time=None,
        status=None,
        custom_properties=None,
        _configuration=None,
    ):  # noqa: E501
        """WebHook - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._web_url = None
        self._links = None
        self._properties = None
        self.__self = None
        self._display_name = None
        self._description = None
        self._events = None
        self._created_date_time = None
        self._last_action_date_time = None
        self._status = None
        self._custom_properties = None
        self.discriminator = None

        self.web_url = web_url
        if links is not None:
            self.links = links
        if properties is not None:
            self.properties = properties
        if _self is not None:
            self._self = _self
        self.display_name = display_name
        if description is not None:
            self.description = description
        self.events = events
        if created_date_time is not None:
            self.created_date_time = created_date_time
        if last_action_date_time is not None:
            self.last_action_date_time = last_action_date_time
        if status is not None:
            self.status = status
        if custom_properties is not None:
            self.custom_properties = custom_properties

    @property
    def web_url(self):
        """Gets the web_url of this WebHook.  # noqa: E501

        The registered URL that will be used to send the POST requests for the registered events to.  # noqa: E501

        :return: The web_url of this WebHook.  # noqa: E501
        :rtype: str
        """
        return self._web_url

    @web_url.setter
    def web_url(self, web_url):
        """Sets the web_url of this WebHook.

        The registered URL that will be used to send the POST requests for the registered events to.  # noqa: E501

        :param web_url: The web_url of this WebHook.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and web_url is None:
            raise ValueError(
                "Invalid value for `web_url`, must not be `None`"
            )  # noqa: E501

        self._web_url = web_url

    @property
    def links(self):
        """Gets the links of this WebHook.  # noqa: E501


        :return: The links of this WebHook.  # noqa: E501
        :rtype: WebHookLinks
        """
        return self._links

    @links.setter
    def links(self, links):
        """Sets the links of this WebHook.


        :param links: The links of this WebHook.  # noqa: E501
        :type: WebHookLinks
        """

        self._links = links

    @property
    def properties(self):
        """Gets the properties of this WebHook.  # noqa: E501


        :return: The properties of this WebHook.  # noqa: E501
        :rtype: WebHookProperties
        """
        return self._properties

    @properties.setter
    def properties(self, properties):
        """Sets the properties of this WebHook.


        :param properties: The properties of this WebHook.  # noqa: E501
        :type: WebHookProperties
        """

        self._properties = properties

    @property
    def _self(self):
        """Gets the _self of this WebHook.  # noqa: E501

        The location of this entity.  # noqa: E501

        :return: The _self of this WebHook.  # noqa: E501
        :rtype: str
        """
        return self.__self

    @_self.setter
    def _self(self, _self):
        """Sets the _self of this WebHook.

        The location of this entity.  # noqa: E501

        :param _self: The _self of this WebHook.  # noqa: E501
        :type: str
        """

        self.__self = _self

    @property
    def display_name(self):
        """Gets the display_name of this WebHook.  # noqa: E501

        The display name of the object.  # noqa: E501

        :return: The display_name of this WebHook.  # noqa: E501
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """Sets the display_name of this WebHook.

        The display name of the object.  # noqa: E501

        :param display_name: The display_name of this WebHook.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and display_name is None:
            raise ValueError(
                "Invalid value for `display_name`, must not be `None`"
            )  # noqa: E501

        self._display_name = display_name

    @property
    def description(self):
        """Gets the description of this WebHook.  # noqa: E501

        The description of the object.  # noqa: E501

        :return: The description of this WebHook.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this WebHook.

        The description of the object.  # noqa: E501

        :param description: The description of this WebHook.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def events(self):
        """Gets the events of this WebHook.  # noqa: E501


        :return: The events of this WebHook.  # noqa: E501
        :rtype: WebHookEvents
        """
        return self._events

    @events.setter
    def events(self, events):
        """Sets the events of this WebHook.


        :param events: The events of this WebHook.  # noqa: E501
        :type: WebHookEvents
        """
        if self._configuration.client_side_validation and events is None:
            raise ValueError(
                "Invalid value for `events`, must not be `None`"
            )  # noqa: E501

        self._events = events

    @property
    def created_date_time(self):
        """Gets the created_date_time of this WebHook.  # noqa: E501

        The time-stamp when the object was created.  The time stamp is encoded as ISO 8601 date and time format  (\"YYYY-MM-DDThh:mm:ssZ\", see https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations).  # noqa: E501

        :return: The created_date_time of this WebHook.  # noqa: E501
        :rtype: datetime
        """
        return self._created_date_time

    @created_date_time.setter
    def created_date_time(self, created_date_time):
        """Sets the created_date_time of this WebHook.

        The time-stamp when the object was created.  The time stamp is encoded as ISO 8601 date and time format  (\"YYYY-MM-DDThh:mm:ssZ\", see https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations).  # noqa: E501

        :param created_date_time: The created_date_time of this WebHook.  # noqa: E501
        :type: datetime
        """

        self._created_date_time = created_date_time

    @property
    def last_action_date_time(self):
        """Gets the last_action_date_time of this WebHook.  # noqa: E501

        The time-stamp when the current status was entered.  The time stamp is encoded as ISO 8601 date and time format  (\"YYYY-MM-DDThh:mm:ssZ\", see https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations).  # noqa: E501

        :return: The last_action_date_time of this WebHook.  # noqa: E501
        :rtype: datetime
        """
        return self._last_action_date_time

    @last_action_date_time.setter
    def last_action_date_time(self, last_action_date_time):
        """Sets the last_action_date_time of this WebHook.

        The time-stamp when the current status was entered.  The time stamp is encoded as ISO 8601 date and time format  (\"YYYY-MM-DDThh:mm:ssZ\", see https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations).  # noqa: E501

        :param last_action_date_time: The last_action_date_time of this WebHook.  # noqa: E501
        :type: datetime
        """

        self._last_action_date_time = last_action_date_time

    @property
    def status(self):
        """Gets the status of this WebHook.  # noqa: E501


        :return: The status of this WebHook.  # noqa: E501
        :rtype: Status
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this WebHook.


        :param status: The status of this WebHook.  # noqa: E501
        :type: Status
        """

        self._status = status

    @property
    def custom_properties(self):
        """Gets the custom_properties of this WebHook.  # noqa: E501

        The custom properties of this entity. The maximum allowed key length is 64 characters, the maximum  allowed value length is 256 characters and the count of allowed entries is 10.  # noqa: E501

        :return: The custom_properties of this WebHook.  # noqa: E501
        :rtype: dict(str, str)
        """
        return self._custom_properties

    @custom_properties.setter
    def custom_properties(self, custom_properties):
        """Sets the custom_properties of this WebHook.

        The custom properties of this entity. The maximum allowed key length is 64 characters, the maximum  allowed value length is 256 characters and the count of allowed entries is 10.  # noqa: E501

        :param custom_properties: The custom_properties of this WebHook.  # noqa: E501
        :type: dict(str, str)
        """

        self._custom_properties = custom_properties

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value)
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value
        if issubclass(WebHook, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, WebHook):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, WebHook):
            return True

        return self.to_dict() != other.to_dict()
