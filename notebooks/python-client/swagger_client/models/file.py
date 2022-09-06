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


class File(object):
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
        "kind": "FileKind",
        "links": "FileLinks",
        "created_date_time": "datetime",
        "properties": "FileProperties",
        "name": "str",
        "_self": "str",
    }

    attribute_map = {
        "kind": "kind",
        "links": "links",
        "created_date_time": "createdDateTime",
        "properties": "properties",
        "name": "name",
        "_self": "self",
    }

    def __init__(
        self,
        kind=None,
        links=None,
        created_date_time=None,
        properties=None,
        name=None,
        _self=None,
        _configuration=None,
    ):  # noqa: E501
        """File - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._kind = None
        self._links = None
        self._created_date_time = None
        self._properties = None
        self._name = None
        self.__self = None
        self.discriminator = None

        if kind is not None:
            self.kind = kind
        if links is not None:
            self.links = links
        if created_date_time is not None:
            self.created_date_time = created_date_time
        if properties is not None:
            self.properties = properties
        if name is not None:
            self.name = name
        if _self is not None:
            self._self = _self

    @property
    def kind(self):
        """Gets the kind of this File.  # noqa: E501


        :return: The kind of this File.  # noqa: E501
        :rtype: FileKind
        """
        return self._kind

    @kind.setter
    def kind(self, kind):
        """Sets the kind of this File.


        :param kind: The kind of this File.  # noqa: E501
        :type: FileKind
        """

        self._kind = kind

    @property
    def links(self):
        """Gets the links of this File.  # noqa: E501


        :return: The links of this File.  # noqa: E501
        :rtype: FileLinks
        """
        return self._links

    @links.setter
    def links(self, links):
        """Sets the links of this File.


        :param links: The links of this File.  # noqa: E501
        :type: FileLinks
        """

        self._links = links

    @property
    def created_date_time(self):
        """Gets the created_date_time of this File.  # noqa: E501

        The creation time of this file.  The time stamp is encoded as ISO 8601 date and time format  (see https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations).  # noqa: E501

        :return: The created_date_time of this File.  # noqa: E501
        :rtype: datetime
        """
        return self._created_date_time

    @created_date_time.setter
    def created_date_time(self, created_date_time):
        """Sets the created_date_time of this File.

        The creation time of this file.  The time stamp is encoded as ISO 8601 date and time format  (see https://en.wikipedia.org/wiki/ISO_8601#Combined_date_and_time_representations).  # noqa: E501

        :param created_date_time: The created_date_time of this File.  # noqa: E501
        :type: datetime
        """

        self._created_date_time = created_date_time

    @property
    def properties(self):
        """Gets the properties of this File.  # noqa: E501


        :return: The properties of this File.  # noqa: E501
        :rtype: FileProperties
        """
        return self._properties

    @properties.setter
    def properties(self, properties):
        """Sets the properties of this File.


        :param properties: The properties of this File.  # noqa: E501
        :type: FileProperties
        """

        self._properties = properties

    @property
    def name(self):
        """Gets the name of this File.  # noqa: E501

        The name of this file.  # noqa: E501

        :return: The name of this File.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this File.

        The name of this file.  # noqa: E501

        :param name: The name of this File.  # noqa: E501
        :type: str
        """

        self._name = name

    @property
    def _self(self):
        """Gets the _self of this File.  # noqa: E501

        The location of this entity.  # noqa: E501

        :return: The _self of this File.  # noqa: E501
        :rtype: str
        """
        return self.__self

    @_self.setter
    def _self(self, _self):
        """Sets the _self of this File.

        The location of this entity.  # noqa: E501

        :param _self: The _self of this File.  # noqa: E501
        :type: str
        """

        self.__self = _self

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
        if issubclass(File, dict):
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
        if not isinstance(other, File):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, File):
            return True

        return self.to_dict() != other.to_dict()
